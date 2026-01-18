from __future__ import absolute_import, division, print_function
import pytest
from django.utils.timezone import now
from awx.main.models import Job
@pytest.mark.django_db
def test_job_wait_successful(run_module, admin_user):
    job = Job.objects.create(status='successful', started=now(), finished=now())
    result = run_module('job_wait', dict(job_id=job.id), admin_user)
    result.pop('invocation', None)
    result['elapsed'] = float(result['elapsed'])
    assert result.pop('finished', '')[:10] == str(job.finished)[:10]
    assert result.pop('started', '')[:10] == str(job.started)[:10]
    assert result == {'status': 'successful', 'changed': False, 'elapsed': job.elapsed, 'id': job.id}