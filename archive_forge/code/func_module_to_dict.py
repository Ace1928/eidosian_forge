from django.core.management.base import BaseCommand
def module_to_dict(module, omittable=lambda k: k.startswith('_') or not k.isupper()):
    """Convert a module namespace to a Python dictionary."""
    return {k: repr(getattr(module, k)) for k in dir(module) if not omittable(k)}