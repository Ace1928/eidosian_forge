from django.contrib.admin.decorators import action, display, register
from django.contrib.admin.filters import (
from django.contrib.admin.options import (
from django.contrib.admin.sites import AdminSite, site
from django.utils.module_loading import autodiscover_modules
def autodiscover():
    autodiscover_modules('admin', register_to=site)