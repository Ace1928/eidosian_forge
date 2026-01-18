from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable
def delete_all_checkpoints(self, path):
    """Delete all checkpoints for the given path."""
    for checkpoint in self.list_checkpoints(path):
        self.delete_checkpoint(checkpoint['id'], path)