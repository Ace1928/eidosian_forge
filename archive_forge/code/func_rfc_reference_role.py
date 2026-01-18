from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def rfc_reference_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    try:
        if '#' in text:
            rfcnum, section = text.split('#', 1)
        else:
            rfcnum, section = (text, None)
        rfcnum = int(rfcnum)
        if rfcnum <= 0:
            raise ValueError
    except ValueError:
        msg = inliner.reporter.error('RFC number must be a number greater than or equal to 1; "%s" is invalid.' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    ref = inliner.document.settings.rfc_base_url + inliner.rfc_url % rfcnum
    if section is not None:
        ref += '#' + section
    set_classes(options)
    node = nodes.reference(rawtext, 'RFC ' + utils.unescape(str(rfcnum)), refuri=ref, **options)
    return ([node], [])