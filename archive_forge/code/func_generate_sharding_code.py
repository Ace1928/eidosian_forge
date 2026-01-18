import hashlib
def generate_sharding_code(standard):
    try:
        combined_string = ''.join((str(standard[key]) for key in standard))
        hash_object = hashlib.sha256(combined_string.encode())
        sharding_code = hash_object.hexdigest()
        return sharding_code
    except Exception as e:
        print(f'Error in generate_sharding_code: {e}')
        raise