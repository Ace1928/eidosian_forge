from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def onStageCompleted(self, stageCompleted):
    from py4j.protocol import Py4JJavaError
    stage_info = stageCompleted.stageInfo()
    message = ''
    level = ''
    data = {'attemptId': stage_info.attemptId(), 'name': stage_info.name()}
    try:
        data['reason'] = stage_info.failureReason().get()
        message = 'Stage {} Failed'.format(stage_info.stageId())
        level = 'warning'
    except Py4JJavaError:
        message = 'Stage {} Completed'.format(stage_info.stageId())
        level = 'info'
    self.hub.add_breadcrumb(level=level, message=message, data=data)