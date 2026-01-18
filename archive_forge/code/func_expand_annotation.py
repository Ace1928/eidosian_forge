from celery.utils.functional import firstmethod, mlazy
from celery.utils.imports import instantiate
def expand_annotation(annotation):
    if isinstance(annotation, dict):
        return MapAnnotation(annotation)
    elif isinstance(annotation, str):
        return mlazy(instantiate, annotation)
    return annotation