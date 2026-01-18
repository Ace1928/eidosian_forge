from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable
def create_notebook_checkpoint(self, nb, path):
    """Create a checkpoint of the current state of a file

        Returns a checkpoint model for the new checkpoint.
        """
    raise NotImplementedError