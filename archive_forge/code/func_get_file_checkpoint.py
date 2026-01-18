from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable
def get_file_checkpoint(self, checkpoint_id, path):
    """Get the content of a checkpoint for a non-notebook file.

        Returns a dict of the form::

            {
                'type': 'file',
                'content': <str>,
                'format': {'text','base64'},
            }
        """
    raise NotImplementedError