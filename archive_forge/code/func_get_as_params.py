def get_as_params(self):
    params = {'QualificationTypeId': self.qualification_type_id, 'Comparator': self.comparator}
    if self.comparator in ('In', 'NotIn'):
        for i, locale in enumerate(self.locale, 1):
            if isinstance(locale, tuple):
                params['LocaleValue.%d.Country' % i] = locale[0]
                params['LocaleValue.%d.Subdivision' % i] = locale[1]
            else:
                params['LocaleValue.%d.Country' % i] = locale
    elif isinstance(self.locale, tuple):
        params['LocaleValue.Country'] = self.locale[0]
        params['LocaleValue.Subdivision'] = self.locale[1]
    else:
        params['LocaleValue.Country'] = self.locale
    if self.required_to_preview:
        params['RequiredToPreview'] = 'true'
    return params