from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def pep_reference_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    try:
        pepnum = int(text)
        if pepnum < 0 or pepnum > 9999:
            raise ValueError
    except ValueError:
        msg = inliner.reporter.error('PEP number must be a number from 0 to 9999; "%s" is invalid.' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    ref = inliner.document.settings.pep_base_url + inliner.document.settings.pep_file_url_template % pepnum
    set_classes(options)
    return ([nodes.reference(rawtext, 'PEP ' + utils.unescape(text), refuri=ref, **options)], [])