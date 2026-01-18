from django.core import checks
def get_cached_value(self, instance, default=NOT_PROVIDED):
    cache_name = self.get_cache_name()
    try:
        return instance._state.fields_cache[cache_name]
    except KeyError:
        if default is NOT_PROVIDED:
            raise
        return default