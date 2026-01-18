import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_parameter(value):
    """ attribute [section] ["*"] [CFWS] "=" value

    The CFWS is implied by the RFC but not made explicit in the BNF.  This
    simplified form of the BNF from the RFC is made to conform with the RFC BNF
    through some extra checks.  We do it this way because it makes both error
    recovery and working with the resulting parse tree easier.
    """
    param = Parameter()
    token, value = get_attribute(value)
    param.append(token)
    if not value or value[0] == ';':
        param.defects.append(errors.InvalidHeaderDefect('Parameter contains name ({}) but no value'.format(token)))
        return (param, value)
    if value[0] == '*':
        try:
            token, value = get_section(value)
            param.sectioned = True
            param.append(token)
        except errors.HeaderParseError:
            pass
        if not value:
            raise errors.HeaderParseError('Incomplete parameter')
        if value[0] == '*':
            param.append(ValueTerminal('*', 'extended-parameter-marker'))
            value = value[1:]
            param.extended = True
    if value[0] != '=':
        raise errors.HeaderParseError("Parameter not followed by '='")
    param.append(ValueTerminal('=', 'parameter-separator'))
    value = value[1:]
    leader = None
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        param.append(token)
    remainder = None
    appendto = param
    if param.extended and value and (value[0] == '"'):
        qstring, remainder = get_quoted_string(value)
        inner_value = qstring.stripped_value
        semi_valid = False
        if param.section_number == 0:
            if inner_value and inner_value[0] == "'":
                semi_valid = True
            else:
                token, rest = get_attrtext(inner_value)
                if rest and rest[0] == "'":
                    semi_valid = True
        else:
            try:
                token, rest = get_extended_attrtext(inner_value)
            except:
                pass
            else:
                if not rest:
                    semi_valid = True
        if semi_valid:
            param.defects.append(errors.InvalidHeaderDefect('Quoted string value for extended parameter is invalid'))
            param.append(qstring)
            for t in qstring:
                if t.token_type == 'bare-quoted-string':
                    t[:] = []
                    appendto = t
                    break
            value = inner_value
        else:
            remainder = None
            param.defects.append(errors.InvalidHeaderDefect('Parameter marked as extended but appears to have a quoted string value that is non-encoded'))
    if value and value[0] == "'":
        token = None
    else:
        token, value = get_value(value)
    if not param.extended or param.section_number > 0:
        if not value or value[0] != "'":
            appendto.append(token)
            if remainder is not None:
                assert not value, value
                value = remainder
            return (param, value)
        param.defects.append(errors.InvalidHeaderDefect('Apparent initial-extended-value but attribute was not marked as extended or was not initial section'))
    if not value:
        param.defects.append(errors.InvalidHeaderDefect('Missing required charset/lang delimiters'))
        appendto.append(token)
        if remainder is None:
            return (param, value)
    else:
        if token is not None:
            for t in token:
                if t.token_type == 'extended-attrtext':
                    break
            t.token_type == 'attrtext'
            appendto.append(t)
            param.charset = t.value
        if value[0] != "'":
            raise errors.HeaderParseError('Expected RFC2231 char/lang encoding delimiter, but found {!r}'.format(value))
        appendto.append(ValueTerminal("'", 'RFC2231-delimiter'))
        value = value[1:]
        if value and value[0] != "'":
            token, value = get_attrtext(value)
            appendto.append(token)
            param.lang = token.value
            if not value or value[0] != "'":
                raise errors.HeaderParseError('Expected RFC2231 char/lang encoding delimiter, but found {}'.format(value))
        appendto.append(ValueTerminal("'", 'RFC2231-delimiter'))
        value = value[1:]
    if remainder is not None:
        v = Value()
        while value:
            if value[0] in WSP:
                token, value = get_fws(value)
            elif value[0] == '"':
                token = ValueTerminal('"', 'DQUOTE')
                value = value[1:]
            else:
                token, value = get_qcontent(value)
            v.append(token)
        token = v
    else:
        token, value = get_value(value)
    appendto.append(token)
    if remainder is not None:
        assert not value, value
        value = remainder
    return (param, value)