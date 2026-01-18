from saharaclient.api import base
def refresh_status(self, obj_id):
    """Refresh Job Status."""
    return self._get('/jobs/%s?refresh_status=True' % obj_id, 'job')