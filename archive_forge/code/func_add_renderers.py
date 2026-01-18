from .compat import escape
from .jsonify import encode
def add_renderers(self, custom_dict):
    """
        Adds a custom renderer.

        :param custom_dict: a dictionary of custom renderers to add
        """
    self._renderer_classes.update(custom_dict)