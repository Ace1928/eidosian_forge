from django.core import checks
def set_cached_value(self, instance, value):
    instance._state.fields_cache[self.get_cache_name()] = value