import os
from django.apps import apps
from django.contrib.staticfiles.finders import get_finders
from django.contrib.staticfiles.storage import staticfiles_storage
from django.core.checks import Tags
from django.core.files.storage import FileSystemStorage
from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.utils.functional import cached_property
def clear_dir(self, path):
    """
        Delete the given relative path using the destination storage backend.
        """
    if not self.storage.exists(path):
        return
    dirs, files = self.storage.listdir(path)
    for f in files:
        fpath = os.path.join(path, f)
        if self.dry_run:
            self.log("Pretending to delete '%s'" % fpath, level=1)
        else:
            self.log("Deleting '%s'" % fpath, level=1)
            try:
                full_path = self.storage.path(fpath)
            except NotImplementedError:
                self.storage.delete(fpath)
            else:
                if not os.path.exists(full_path) and os.path.lexists(full_path):
                    os.unlink(full_path)
                else:
                    self.storage.delete(fpath)
    for d in dirs:
        self.clear_dir(os.path.join(path, d))