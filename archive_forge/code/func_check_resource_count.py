def check_resource_count(expected_count):
    test.assertEqual(expected_count, len(reality.all_resources()))