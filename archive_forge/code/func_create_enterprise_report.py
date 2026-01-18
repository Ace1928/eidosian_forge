from .util import coalesce
def create_enterprise_report(project=None, title='Untitled Report', description='', header=None, body=None, footer=None):
    """Create an example enterprise report with a header and footer.

    Can be used to add custom branding to reports.
    """
    import wandb.apis.reports as wr
    project = coalesce(project, 'default-project')
    header = coalesce(header, create_example_header())
    body = coalesce(body, [])
    footer = coalesce(footer, create_example_footer())
    return wr.Report(project=project, title=title, description=description, blocks=[*header, *body, *footer])