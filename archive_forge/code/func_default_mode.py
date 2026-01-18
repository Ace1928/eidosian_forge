from pprint import pformat
from six import iteritems
import re
@default_mode.setter
def default_mode(self, default_mode):
    """
        Sets the default_mode of this V1DownwardAPIVolumeSource.
        Optional: mode bits to use on created files by default. Must be a value
        between 0 and 0777. Defaults to 0644. Directories within the path are
        not affected by this setting. This might be in conflict with other
        options that affect the file mode, like fsGroup, and the result can be
        other mode bits set.

        :param default_mode: The default_mode of this V1DownwardAPIVolumeSource.
        :type: int
        """
    self._default_mode = default_mode