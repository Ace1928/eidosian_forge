def check_c_count(expected_count):
    test.assertEqual(expected_count, len(reality.resources_by_logical_name('C')))