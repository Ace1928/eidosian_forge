from io import StringIO
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
def deserialize_m2m_values(field, field_value, using, handle_forward_references):
    model = field.remote_field.model
    if hasattr(model._default_manager, 'get_by_natural_key'):

        def m2m_convert(value):
            if hasattr(value, '__iter__') and (not isinstance(value, str)):
                return model._default_manager.db_manager(using).get_by_natural_key(*value).pk
            else:
                return model._meta.pk.to_python(value)
    else:

        def m2m_convert(v):
            return model._meta.pk.to_python(v)
    try:
        pks_iter = iter(field_value)
    except TypeError as e:
        raise M2MDeserializationError(e, field_value)
    try:
        values = []
        for pk in pks_iter:
            values.append(m2m_convert(pk))
        return values
    except Exception as e:
        if isinstance(e, ObjectDoesNotExist) and handle_forward_references:
            return DEFER_FIELD
        else:
            raise M2MDeserializationError(e, pk)