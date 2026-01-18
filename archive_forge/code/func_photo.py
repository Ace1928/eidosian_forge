from . import mf2_classes
from .dom_helpers import get_attr, get_children, get_img, get_textContent, try_urljoin
def photo(el, base_url, filtered_roots):
    """Find an implied photo property

    Args:
      el (bs4.element.Tag): a DOM element
      base_url (string): the base URL to use, to reconcile relative URLs

    Returns:
      string or dictionary: the implied photo value or implied photo as a dictionary with alt value
    """

    def get_photo_child(children):
        """take a list of children and finds a valid child for photo property"""
        poss_imgs = [c for c in children if c.name == 'img']
        if len(poss_imgs) == 1:
            poss_img = poss_imgs[0]
            if not mf2_classes.root(poss_img.get('class', []), filtered_roots):
                return poss_img
        poss_objs = [c for c in children if c.name == 'object']
        if len(poss_objs) == 1:
            poss_obj = poss_objs[0]
            if not mf2_classes.root(poss_obj.get('class', []), filtered_roots):
                return poss_obj

    def resolve_relative_url(prop_value):
        if isinstance(prop_value, dict):
            prop_value['value'] = try_urljoin(base_url, prop_value['value'])
        else:
            prop_value = try_urljoin(base_url, prop_value)
        return prop_value
    if (prop_value := get_img(el, base_url)):
        return resolve_relative_url(prop_value)
    if (prop_value := get_attr(el, 'data', check_name='object')):
        return resolve_relative_url(prop_value)
    poss_child = None
    children = list(get_children(el))
    poss_child = get_photo_child(children)
    if poss_child is None and len(children) == 1 and (not mf2_classes.root(children[0].get('class', []), filtered_roots)):
        grandchildren = list(get_children(children[0]))
        poss_child = get_photo_child(grandchildren)
    if poss_child is not None:
        if (prop_value := get_img(poss_child, base_url)):
            return resolve_relative_url(prop_value)
        if (prop_value := get_attr(poss_child, 'data', check_name='object')):
            return resolve_relative_url(prop_value)