from django.http import HttpResponse
from .loader import get_template, select_template
def resolve_context(self, context):
    return context