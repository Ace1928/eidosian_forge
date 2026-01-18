from django.core.exceptions import ImproperlyConfigured, SuspiciousFileOperation
from django.template.utils import get_app_template_dirs
from django.utils._os import safe_join
from django.utils.functional import cached_property
def iter_template_filenames(self, template_name):
    """
        Iterate over candidate files for template_name.

        Ignore files that don't lie inside configured template dirs to avoid
        directory traversal attacks.
        """
    for template_dir in self.template_dirs:
        try:
            yield safe_join(template_dir, template_name)
        except SuspiciousFileOperation:
            pass