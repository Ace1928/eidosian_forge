import warnings
def squelch_warnings(insecure_requests=True):
    if SubjectAltNameWarning:
        warnings.filterwarnings('ignore', category=SubjectAltNameWarning)
    if InsecurePlatformWarning:
        warnings.filterwarnings('ignore', category=InsecurePlatformWarning)
    if SNIMissingWarning:
        warnings.filterwarnings('ignore', category=SNIMissingWarning)
    if insecure_requests and InsecureRequestWarning:
        warnings.filterwarnings('ignore', category=InsecureRequestWarning)