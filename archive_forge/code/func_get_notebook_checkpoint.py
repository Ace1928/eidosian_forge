from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable
def get_notebook_checkpoint(self, checkpoint_id, path):
    """Get the content of a checkpoint for a notebook.

        Returns a dict of the form::

            {
                'type': 'notebook',
                'content': <output of nbformat.read>,
            }
        """
    raise NotImplementedError