def check_only(self, obj):
    """Run checks on `obj` returning reports

        Parameters
        ----------
        obj : anything
           object on which to run checks

        Returns
        -------
        reports : sequence
           sequence of report objects reporting on result of running
           checks (without fixes) on `obj`
        """
    reports = []
    for check in self._checks:
        obj, rep = check(obj, False)
        reports.append(rep)
    return reports