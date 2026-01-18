import inspect
import sys
def for_code(error_code):
    return kafka_errors.get(error_code, UnknownError)