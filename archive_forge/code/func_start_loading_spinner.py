from ..config import config
def start_loading_spinner(*objects):
    """
    Changes the appearance of the specified panel objects to indicate
    that they are loading.

    This is done by

    * adding a small spinner on top
    * graying out the panel
    * disabling the panel
    * and changing the mouse cursor to a spinner when hovering over the panel

    Arguments
    ---------
    objects: tuple
        The panels to add the loading indicator to.
    """
    css_classes = [LOADING_INDICATOR_CSS_CLASS, f'pn-{config.loading_spinner}']
    for item in objects:
        if hasattr(item, 'css_classes'):
            _add_css_classes(item, css_classes)