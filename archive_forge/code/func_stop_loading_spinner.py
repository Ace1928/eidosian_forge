from ..config import config
def stop_loading_spinner(*objects):
    """
    Removes the loading indicating from the specified panel objects.

    Arguments
    ---------
    objects: tuple
        The panels to remove the loading indicator from.
    """
    css_classes = [LOADING_INDICATOR_CSS_CLASS, f'pn-{config.loading_spinner}']
    for item in objects:
        if hasattr(item, 'css_classes'):
            _remove_css_classes(item, css_classes)