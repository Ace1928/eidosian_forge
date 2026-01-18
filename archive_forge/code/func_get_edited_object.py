import json
from django.conf import settings
from django.contrib.admin.utils import quote
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.urls import NoReverseMatch, reverse
from django.utils import timezone
from django.utils.text import get_text_list
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def get_edited_object(self):
    """Return the edited object represented by this log entry."""
    return self.content_type.get_object_for_this_type(pk=self.object_id)