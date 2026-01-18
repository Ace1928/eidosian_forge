import dill
def test_doc_dill_issue_219():
    back_fn = dill.loads(dill.dumps(get_fun_with_strftime()))
    assert back_fn() == '1943-01-04 00:00:00'
    dupl = dill.loads(dill.dumps(get_fun_with_strftime2))
    assert dupl() == get_fun_with_strftime2()