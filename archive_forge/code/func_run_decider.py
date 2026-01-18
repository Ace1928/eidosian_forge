import time
import uuid
import json
import traceback
from boto.swf.layer1_decisions import Layer1Decisions
from tests.integration.swf.test_layer1 import SimpleWorkflowLayer1TestBase
def run_decider(self):
    """
        run one iteration of a simple decision engine
        """
    tries = 0
    while True:
        dtask = self.conn.poll_for_decision_task(self._domain, self._task_list, reverse_order=True)
        if dtask.get('taskToken') is not None:
            break
        time.sleep(2)
        tries += 1
        if tries > 10:
            assert False, 'no decision task occurred'
    ignorable = ('DecisionTaskScheduled', 'DecisionTaskStarted', 'DecisionTaskTimedOut')
    event = None
    for tevent in dtask['events']:
        if tevent['eventType'] not in ignorable:
            event = tevent
            break
    decisions = Layer1Decisions()
    if event['eventType'] == 'WorkflowExecutionStarted':
        activity_id = str(uuid.uuid1())
        decisions.schedule_activity_task(activity_id, self._activity_type_name, self._activity_type_version, task_list=self._task_list, input=event['workflowExecutionStartedEventAttributes']['input'])
    elif event['eventType'] == 'ActivityTaskCompleted':
        decisions.complete_workflow_execution(result=event['activityTaskCompletedEventAttributes']['result'])
    elif event['eventType'] == 'ActivityTaskFailed':
        decisions.fail_workflow_execution(reason=event['activityTaskFailedEventAttributes']['reason'], details=event['activityTaskFailedEventAttributes']['details'])
    else:
        decisions.fail_workflow_execution(reason='unhandled decision task type; %r' % (event['eventType'],))
    r = self.conn.respond_decision_task_completed(dtask['taskToken'], decisions=decisions._data, execution_context=None)
    assert r is None