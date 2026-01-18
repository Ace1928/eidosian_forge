from .jsonutil import JsonTable
def search_experiments(self, project_id=None, subject_id=None, subject_label=None, experiment_type='xnat:subjectAssessorData', columns=None, constraints=None):
    """ Returns a list of all visible experiment IDs of the
            specified type, filtered by optional constraints. This
            function is a shortcut using the search engine.

            Parameters
            ----------
            project_id: string
                Name pattern to filter by project ID.
            subject_id: string
                Name pattern to filter by subject ID.
            subject_label: string
                Name pattern to filter by subject ID.
            experiment_type: string
                xsi path type must be a leaf session type.
                defaults to 'xnat:mrSessionData'
            columns: List[string]
                list of xsi paths for names of columns to return.
            constraints: list[(tupple)]
                List of tupples for comparison in the form (key, comparison,
                value) valid comparisons are: =, <, <=,>,>=, LIKE
            """
    if columns is None:
        columns = []
    where_clause = []
    if project_id is not None:
        item = ('%s/project' % experiment_type, '=', project_id)
        where_clause.append(item)
    if subject_id is not None:
        where_clause.append(('xnat:subjectData/ID', '=', subject_id))
    if subject_label is not None:
        where_clause.append(('xnat:subjectData/LABEL', '=', subject_label))
    if constraints is not None:
        where_clause.extend(constraints)
    if where_clause != []:
        where_clause.append('AND')
    if where_clause != []:
        sel = self._intf.select(experiment_type, columns=columns)
        table = sel.where(where_clause)
        return table
    else:
        table = self._intf.select(experiment_type, columns=columns)
        return table.all()