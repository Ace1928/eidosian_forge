import ssl
def pending(self):
    return self._fetcher is not None and self._fetcher.pending()