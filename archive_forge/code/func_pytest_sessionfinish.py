import pytest
def pytest_sessionfinish(session, exitstatus):
    from pyannotate_runtime import collect_types
    collect_types.dump_stats('type_info.json')