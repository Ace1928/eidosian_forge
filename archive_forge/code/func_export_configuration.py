import json
import os
from flask import (
from flask_bootstrap import Bootstrap
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from nbconvert import HTMLExporter
import nbformat
import subprocess
import jupyterlab
import logging
from flask import request
import bcrypt
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import IntegrityError
@app.route('/export_config', methods=['GET'])
def export_configuration() -> Any:
    """
    Export the current configuration to a downloadable file.
    This endpoint meticulously provides a method for the client to download the current configuration as a JSON file, thereby facilitating the backup and portability of settings with the utmost precision and attention to detail. It is designed to ensure that the configuration is exported accurately and efficiently, reflecting the current state of the application settings comprehensively.

    Returns:
        Any: A meticulously crafted downloadable JSON file containing the current configuration, structured to ensure maximum utility and adherence to the highest standards of data integrity and user experience.
    """
    try:
        gui_handler = UniversalGUI()
        current_configuration = gui_handler.load_config()
        if current_configuration is None:
            raise ValueError('The current configuration could not be retrieved from the system. Please ensure the system is properly configured and the configuration file is accessible.')
        json_formatted_configuration = jsonify(current_configuration)
        json_formatted_configuration.headers['Content-Disposition'] = 'attachment; filename=configuration.json'
        app.logger.info(f'Configuration file prepared for download with the current settings: {current_configuration}')
        return json_formatted_configuration
    except Exception as e:
        app.logger.error(f'Failed to export configuration due to: {type(e).__name__}: {str(e)}')
        error_response = jsonify({'error': f'Failed to export configuration due to: {type(e).__name__}: {str(e)}', 'success': False})
        error_response.status_code = 500
        return error_response