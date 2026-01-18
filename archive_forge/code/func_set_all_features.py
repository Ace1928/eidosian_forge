from ._Feature import Feature
import re
def set_all_features(self, attr, value):
    """Set an attribute of all the features.

        Arguments:
         - attr: An attribute of the Feature class
         - value: The value to set that attribute to

        Set the passed attribute of all features in the set to the
        passed value.
        """
    for feature in self.features.values():
        if hasattr(feature, attr):
            setattr(feature, attr, value)