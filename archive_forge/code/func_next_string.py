import ssl
def next_string(self):
    if self._fetcher is not None and (not self._fetcher.done()):
        raise InappropriateParserState('next_string() invoked while previous fetcher is not exhausted')
    self._fetcher = SingleNetstringFetcher(self._incoming, self._maxlen)
    return self._fetcher