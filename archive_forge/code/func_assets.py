import abc
from tensorflow.python.framework import ops
@abc.abstractmethod
def assets(self):
    """Provide all of the assets contained by the PluginAsset instance.

    The assets method should return a dictionary structured as
    {asset_name: asset_contents}. asset_contents is a string.

    This method will be called by the tf.compat.v1.summary.FileWriter when it
    is time to write the assets out to disk.
    """
    raise NotImplementedError()