import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
def ofp_instruction_from_str(ofproto, action_str):
    """
    Parse an ovs-ofctl style action string and return a list of
    jsondict representations of OFPInstructionActions, which
    can then be passed to ofproto_parser.ofp_instruction_from_jsondict.

    Please note that this is for making transition from ovs-ofctl
    easier. Please consider using OFPAction constructors when writing
    new codes.

    This function takes the following arguments.

    =========== =================================================
    Argument    Description
    =========== =================================================
    ofproto     An ofproto module.
    action_str  An action string.
    =========== =================================================
    """
    action_re = re.compile('([a-z_]+)(\\([^)]*\\)|[^a-z_,()][^,()]*)*')
    result = []
    while len(action_str):
        m = action_re.match(action_str)
        if not m:
            raise os_ken.exception.OFPInvalidActionString(action_str=action_str)
        action_name = m.group(1)
        this_action = m.group(0)
        paren_level = this_action.count('(') - this_action.count(')')
        assert paren_level >= 0
        try:
            if paren_level > 0:
                this_action, rest = _tokenize_paren_block(action_str, m.end(0))
            else:
                rest = action_str[m.end(0):]
            if len(rest):
                assert rest[0] == ','
                rest = rest[1:]
        except Exception:
            raise os_ken.exception.OFPInvalidActionString(action_str=action_str)
        if action_name == 'drop':
            assert this_action == 'drop'
            assert len(result) == 0 and rest == ''
            return []
        converter = getattr(OfctlActionConverter, action_name, None)
        if converter is None or not callable(converter):
            raise os_ken.exception.OFPInvalidActionString(action_str=action_name)
        result.append(converter(ofproto, this_action))
        action_str = rest
    return result