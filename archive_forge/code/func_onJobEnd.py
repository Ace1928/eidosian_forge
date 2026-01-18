from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def onJobEnd(self, jobEnd):
    level = ''
    message = ''
    data = {'result': jobEnd.jobResult().toString()}
    if jobEnd.jobResult().toString() == 'JobSucceeded':
        level = 'info'
        message = 'Job {} Ended'.format(jobEnd.jobId())
    else:
        level = 'warning'
        message = 'Job {} Failed'.format(jobEnd.jobId())
    self.hub.add_breadcrumb(level=level, message=message, data=data)