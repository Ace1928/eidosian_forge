import json
import os
import re
from pathlib import Path
from django.apps import apps
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import Context, Engine
from django.urls import translate_url
from django.utils.formats import get_format
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import check_for_language, get_language
from django.utils.translation.trans_real import DjangoTranslation
from django.views.generic import View
def get_formats():
    """Return all formats strings required for i18n to work."""
    FORMAT_SETTINGS = ('DATE_FORMAT', 'DATETIME_FORMAT', 'TIME_FORMAT', 'YEAR_MONTH_FORMAT', 'MONTH_DAY_FORMAT', 'SHORT_DATE_FORMAT', 'SHORT_DATETIME_FORMAT', 'FIRST_DAY_OF_WEEK', 'DECIMAL_SEPARATOR', 'THOUSAND_SEPARATOR', 'NUMBER_GROUPING', 'DATE_INPUT_FORMATS', 'TIME_INPUT_FORMATS', 'DATETIME_INPUT_FORMATS')
    return {attr: get_format(attr) for attr in FORMAT_SETTINGS}