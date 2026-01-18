import warnings
from IPython.core.getipython import get_ipython
def install_payload_page():
    """DEPRECATED, use show_in_pager hook

    Install this version of page as IPython.core.page.page.
    """
    warnings.warn("install_payload_page is deprecated.\n    Use `ip.set_hook('show_in_pager, page.as_hook(payloadpage.page))`\n    ")
    from IPython.core import page as corepage
    corepage.page = page