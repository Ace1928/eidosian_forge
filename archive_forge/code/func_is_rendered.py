from django.http import HttpResponse
from .loader import get_template, select_template
@property
def is_rendered(self):
    return self._is_rendered