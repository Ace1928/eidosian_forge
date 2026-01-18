import requests
def is_client_error(self):
    if self.status_code is None:
        return False
    return 400 <= self.status_code < 500