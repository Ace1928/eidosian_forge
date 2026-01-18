import sys
from django.core.management.color import color_style
from django.db import IntegrityError, migrations, transaction
from django.db.models import Q
def revert_proxy_model_permissions(apps, schema_editor):
    """
    Update the content_type of proxy model permissions to use the ContentType
    of the concrete model.
    """
    update_proxy_model_permissions(apps, schema_editor, reverse=True)