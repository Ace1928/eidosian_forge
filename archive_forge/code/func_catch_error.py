import re
from lxml import etree
def catch_error(msg_or_exception, full_response=None):
    if isinstance(msg_or_exception, bytes):
        msg_or_exception = msg_or_exception.decode()
    if isinstance(msg_or_exception, str):
        msg = msg_or_exception
        error = parse_error_message(msg)
        if error == 'The request requires user authentication':
            raise OperationalError('Authentication failed')
        elif 'Not Found' in error:
            raise OperationalError('Connection failed')
        elif full_response:
            raise DatabaseError(full_response)
        else:
            raise DatabaseError(error)
    else:
        raise DatabaseError(str(msg_or_exception))