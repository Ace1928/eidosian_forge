def is_botocore_available() -> bool:
    try:
        import botocore
        return True
    except ImportError:
        return False