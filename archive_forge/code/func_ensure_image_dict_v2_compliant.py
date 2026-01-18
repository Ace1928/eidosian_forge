from glance.common import exception
from glance.i18n import _
def ensure_image_dict_v2_compliant(image):
    """
    Accepts an image dictionary that contains a v1-style 'is_public' member
    and returns the equivalent v2-style image dictionary.
    """
    if 'is_public' in image:
        if 'visibility' in image:
            msg = _("Specifying both 'visibility' and 'is_public' is not permiitted.")
            raise exception.Invalid(msg)
        else:
            image['visibility'] = 'public' if image.pop('is_public') else 'shared'
    return image