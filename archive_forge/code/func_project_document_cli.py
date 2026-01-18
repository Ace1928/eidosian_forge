from pathlib import Path
from wasabi import MarkdownRenderer, msg
from ..util import load_project_config, working_dir
from .main import PROJECT_FILE, Arg, Opt, app
@app.command('document')
def project_document_cli(project_dir: Path=Arg(Path.cwd(), help='Path to cloned project. Defaults to current working directory.', exists=True, file_okay=False), output_file: Path=Opt('-', '--output', '-o', help='Path to output Markdown file for output. Defaults to - for standard output'), no_emoji: bool=Opt(False, '--no-emoji', '-NE', help="Don't use emoji")):
    """
    Auto-generate a README.md for a project. If the content is saved to a file,
    hidden markers are added so you can add custom content before or after the
    auto-generated section and only the auto-generated docs will be replaced
    when you re-run the command.

    DOCS: https://github.com/explosion/weasel/tree/main/docs/cli.md#closed_book-document
    """
    project_document(project_dir, output_file, no_emoji=no_emoji)