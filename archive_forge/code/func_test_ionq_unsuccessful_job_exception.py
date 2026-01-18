import cirq_ionq as ionq
def test_ionq_unsuccessful_job_exception():
    ex = ionq.IonQUnsuccessfulJobException(job_id='SWE', status='canceled')
    assert str(ex) == "Status code: None, Message: 'Job SWE was canceled.'"
    assert ex.status_code is None