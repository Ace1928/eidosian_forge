def test_certs(self):
    self.assertTrue(len(self.regions) > 0)
    for region in self.regions:
        special_access_required = False
        for snippet in ('gov', 'cn-'):
            if snippet in region.name:
                special_access_required = True
                break
        try:
            c = region.connect()
            self.sample_service_call(c)
        except:
            if not special_access_required:
                raise