import json
import uuid
import hashlib
import logging
def generate_registry_code(standard):
    try:
        unique_str = str(uuid.uuid4()) + str(standard)
        hash_object = hashlib.sha256(unique_str.encode())
        return hash_object.hexdigest()
    except Exception as e:
        logging.error(f'Error in generate_registry_code: {e}')
        raise