from uc_micro.categories import Cc, Cf, P, Z
from uc_micro.properties import Any
def build_re(opts):
    """Build regex

    Args:
        opts (dict): options

    Return:
        dict: dict of regex string
    """
    SRC_HOST_STRICT = SRC_HOST + _re_host_terminator(opts)
    TPL_HOST_FUZZY_STRICT = TPL_HOST_FUZZY + _re_host_terminator(opts)
    SRC_HOST_PORT_STRICT = SRC_HOST + SRC_PORT + _re_host_terminator(opts)
    TPL_HOST_PORT_FUZZY_STRICT = TPL_HOST_FUZZY + SRC_PORT + _re_host_terminator(opts)
    TPL_HOST_PORT_NO_IP_FUZZY_STRICT = TPL_HOST_NO_IP_FUZZY + SRC_PORT + _re_host_terminator(opts)
    TPL_EMAIL_FUZZY = '(^|' + TEXT_SEPARATORS + '|"|\\(|' + SRC_ZCC + ')' + '(' + SRC_EMAIL_NAME + '@' + TPL_HOST_FUZZY_STRICT + ')'
    regex = {'src_Any': SRC_ANY, 'src_Cc': SRC_CC, 'src_Cf': SRC_CF, 'src_Z': SRC_Z, 'src_P': SRC_P, 'src_ZPCc': SRC_ZPCC, 'src_ZCc': SRC_ZCC, 'src_pseudo_letter': SRC_PSEUDO_LETTER, 'src_ip4': SRC_IP4, 'src_auth': SRC_AUTH, 'src_port': SRC_PORT, 'src_host_terminator': _re_host_terminator(opts), 'src_path': _re_src_path(opts), 'src_email_name': SRC_EMAIL_NAME, 'src_xn': SRC_XN, 'src_domain_root': SRC_DOMAIN_ROOT, 'src_domain': SRC_DOMAIN, 'src_host': SRC_HOST, 'tpl_host_fuzzy': TPL_HOST_FUZZY, 'tpl_host_no_ip_fuzzy': TPL_HOST_NO_IP_FUZZY, 'src_host_strict': SRC_HOST_STRICT, 'tpl_host_fuzzy_strict': TPL_HOST_FUZZY_STRICT, 'src_host_port_strict': SRC_HOST_PORT_STRICT, 'tpl_host_port_fuzzy_strict': TPL_HOST_PORT_FUZZY_STRICT, 'tpl_host_port_no_ip_fuzzy_strict': TPL_HOST_PORT_FUZZY_STRICT, 'tpl_host_fuzzy_test': TPL_HOST_FUZZY_TEST, 'tpl_email_fuzzy': TPL_EMAIL_FUZZY, 'tpl_link_fuzzy': '(^|(?![.:/\\-_@])(?:[$+<=>^`|｜]|' + SRC_ZPCC + '))' + '((?![$+<=>^`|｜])' + TPL_HOST_PORT_FUZZY_STRICT + _re_src_path(opts) + ')', 'tpl_link_no_ip_fuzzy': '(^|(?![.:/\\-_@])(?:[$+<=>^`|｜]|' + SRC_ZPCC + '))' + '((?![$+<=>^`|｜])' + TPL_HOST_PORT_NO_IP_FUZZY_STRICT + _re_src_path(opts) + ')'}
    return regex