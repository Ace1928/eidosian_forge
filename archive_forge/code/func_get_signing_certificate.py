def get_signing_certificate(self):
    """Get signing certificate.

        :returns: PEM-formatted string.
        :rtype: str

        """
    resp, body = self._client.get('/certificates/signing', authenticated=False)
    return resp.text